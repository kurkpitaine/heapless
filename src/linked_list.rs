//! A fixed linked list, more classical than [`SortedLinkedList`].
//!
//! # Examples
//!
//! ```
//! use heapless::linked_list::LinkedList;
//! let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
//!
//! ll.push_front(1).unwrap();
//! assert_eq!(ll.front(), Some(&1));
//!
//! ll.push_front(2).unwrap();
//! assert_eq!(ll.front(), Some(&2));
//!
//! ll.push_front(3).unwrap();
//! assert_eq!(ll.front(), Some(&3));
//!
//! // This will not fit in the queue.
//! assert_eq!(ll.push_front(4), Err(4));
//! ```
//!
//! [`SortedLinkedList`]: `crate::linked_list::SortedLinkedList`

use core::fmt;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::ptr;

/// Trait for defining an index for the linked list, never implemented by users.
pub trait LinkedListIndex: Copy + PartialEq {
    #[doc(hidden)]
    unsafe fn new_unchecked(val: usize) -> Self;
    #[doc(hidden)]
    unsafe fn get_unchecked(self) -> usize;
    #[doc(hidden)]
    fn option(self) -> Option<usize>;
    #[doc(hidden)]
    fn none() -> Self;
}

/// Sealed traits
mod private {
    pub trait Sealed {}
}

/// A node in the [`LinkedList`].
#[derive(Debug)]
pub struct Node<T, Idx> {
    val: MaybeUninit<T>,
    prev: Idx,
    next: Idx,
}

/// The linked list.
pub struct LinkedList<T, Idx, const N: usize>
where
    Idx: LinkedListIndex,
{
    list: [Node<T, Idx>; N],
    len: usize,
    head: Idx,
    tail: Idx,
    free: Idx,
}

// Internal macro for generating indexes for the linkedlist and const new for the linked list
macro_rules! impl_index_and_const_new {
    ($name:ident, $ty:ty, $new_name:ident, $max_val:expr) => {
        /// Index for the [`LinkedList`] with specific backing storage.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name($ty);

        impl LinkedListIndex for $name {
            #[inline(always)]
            unsafe fn new_unchecked(val: usize) -> Self {
                Self::new_unchecked(val as $ty)
            }

            /// This is only valid if `self.option()` is not `None`.
            #[inline(always)]
            unsafe fn get_unchecked(self) -> usize {
                self.0 as usize
            }

            #[inline(always)]
            fn option(self) -> Option<usize> {
                if self.0 == <$ty>::MAX {
                    None
                } else {
                    Some(self.0 as usize)
                }
            }

            #[inline(always)]
            fn none() -> Self {
                Self::none()
            }
        }

        impl $name {
            /// Needed for a `const fn new()`.
            #[inline]
            const unsafe fn new_unchecked(value: $ty) -> Self {
                $name(value)
            }

            /// Needed for a `const fn new()`.
            #[inline]
            const fn none() -> Self {
                $name(<$ty>::MAX)
            }
        }

        impl<T, const N: usize> LinkedList<T, $name, N> {
            const UNINIT: Node<T, $name> = Node {
                val: MaybeUninit::uninit(),
                prev: $name::none(),
                next: $name::none(),
            };

            /// Create a new linked list.
            pub const fn $new_name() -> Self {
                // Const assert N < MAX
                crate::sealed::smaller_than::<N, $max_val>();

                let mut list = LinkedList {
                    list: [Self::UNINIT; N],
                    len: 0,
                    head: $name::none(),
                    tail: $name::none(),
                    free: unsafe { $name::new_unchecked(0) },
                };

                if N == 0 {
                    list.free = $name::none();
                    return list;
                }

                // Initialize indexes
                // Manually initialize head
                list.list[0].next = unsafe { $name::new_unchecked(0 as $ty + 1) };

                let mut free = 1;
                while free < N - 1 {
                    list.list[free].prev = unsafe { $name::new_unchecked(free as $ty - 1) };
                    list.list[free].next = unsafe { $name::new_unchecked(free as $ty + 1) };
                    free += 1;
                }

                // Manually initialize tail
                list.list[free].prev = unsafe { $name::new_unchecked(free as $ty - 1) };

                list
            }

            /// Returns the maximum number of elements the linked list can hold.
            pub const fn capacity(&self) -> usize {
                N
            }

            /// Returns the number of elements currently in the linked list.
            pub fn len(&self) -> usize {
                self.len
            }
        }
    };
}

impl_index_and_const_new!(LinkedIndexU8, u8, new_u8, { u8::MAX as usize - 1 });
impl_index_and_const_new!(LinkedIndexU16, u16, new_u16, { u16::MAX as usize - 1 });
impl_index_and_const_new!(LinkedIndexUsize, usize, new_usize, { usize::MAX - 1 });

impl<T, Idx, const N: usize> LinkedList<T, Idx, N>
where
    Idx: LinkedListIndex,
{
    /// Internal access helper
    #[inline(always)]
    fn node_at(&self, index: usize) -> &Node<T, Idx> {
        // Safety: The entire `self.list` is initialized in `new`, which makes this safe.
        unsafe { self.list.get_unchecked(index) }
    }

    /// Internal access helper
    #[inline(always)]
    fn node_at_mut(&mut self, index: usize) -> &mut Node<T, Idx> {
        // Safety: The entire `self.list` is initialized in `new`, which makes this safe.
        unsafe { self.list.get_unchecked_mut(index) }
    }

    /// Internal access helper
    #[inline(always)]
    fn write_data_in_node_at(&mut self, index: usize, data: T) {
        // Safety: The entire `self.list` is initialized in `new`, which makes this safe.
        unsafe {
            self.node_at_mut(index).val.as_mut_ptr().write(data);
        }
    }

    /// Internal access helper
    #[inline(always)]
    fn read_data_in_node_at(&self, index: usize) -> &T {
        // Safety: The entire `self.list` is initialized in `new`, which makes this safe.
        unsafe { &*self.node_at(index).val.as_ptr() }
    }

    /// Internal access helper
    #[inline(always)]
    fn read_mut_data_in_node_at(&mut self, index: usize) -> &mut T {
        // Safety: The entire `self.list` is initialized in `new`, which makes this safe.
        unsafe { &mut *self.node_at_mut(index).val.as_mut_ptr() }
    }

    /// Internal access helper
    #[inline(always)]
    fn extract_data_in_node_at(&mut self, index: usize) -> T {
        // Safety: The entire `self.list` is initialized in `new`, which makes this safe.
        unsafe { self.node_at(index).val.as_ptr().read() }
    }
}

impl<T, Idx, const N: usize> LinkedList<T, Idx, N>
where
    Idx: LinkedListIndex,
{
    /// Pushes a value at the beginning of the list without checking if the list is full.
    ///
    /// Complexity is worst-case `O(1)`.
    ///
    /// # Safety
    ///
    /// Assumes that the list is not full.
    pub unsafe fn push_front_unchecked(&mut self, value: T) {
        let new = self.free.get_unchecked();

        // Store the data and update the next free spot
        self.write_data_in_node_at(new, value);
        self.free = self.node_at(new).next;

        // Increment len
        self.len += 1;

        // Update list second element, which is previous head.
        if let Some(head) = self.head.option() {
            self.node_at_mut(head).prev = Idx::new_unchecked(new);
            self.node_at_mut(new).next = self.head;
        } else {
            // The list is empty, head next node is None.
            self.node_at_mut(new).next = Idx::none();
        }

        // Previous node on head is always None.
        self.node_at_mut(new).prev = Idx::none();

        // Update head index - new value is the new head.
        self.head = Idx::new_unchecked(new);

        // Update tail index - only at first insertion.
        if self.tail.option().is_none() {
            self.tail = self.head;
        }
    }

    /// Pushes an element at the beginning of the linked list.
    ///
    /// Complexity is `O(1)`.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// // The largest value will always be first
    /// ll.push_front(1).unwrap();
    /// assert_eq!(ll.front(), Some(&1));
    ///
    /// ll.push_front(2).unwrap();
    /// assert_eq!(ll.front(), Some(&2));
    ///
    /// ll.push_front(3).unwrap();
    /// assert_eq!(ll.front(), Some(&3));
    ///
    /// // This will not fit in the queue.
    /// assert_eq!(ll.push_front(4), Err(4));
    /// ```
    pub fn push_front(&mut self, value: T) -> Result<(), T> {
        if !self.is_full() {
            Ok(unsafe {
                self.push_front_unchecked(value);
            })
        } else {
            Err(value)
        }
    }

    /// Pushes a value at the end of the list without checking if the list is full.
    ///
    /// Complexity is worst-case `O(1)`.
    ///
    /// # Safety
    ///
    /// Assumes that the list is not full.
    pub unsafe fn push_back_unchecked(&mut self, value: T) {
        let new = self.free.get_unchecked();

        // Store the data and update the next free spot
        self.write_data_in_node_at(new, value);
        self.free = self.node_at(new).next;

        // Increment len
        self.len += 1;

        // Update list pre-last element, which is previous tail.
        if let Some(tail) = self.tail.option() {
            self.node_at_mut(tail).next = Idx::new_unchecked(new);
            self.node_at_mut(new).prev = self.tail;
        } else {
            // The list is empty, tail prev node is None.
            self.node_at_mut(new).prev = Idx::none();
        }

        // Next node on tail is always None.
        self.node_at_mut(new).next = Idx::none();

        // Update tail index - new value is the new tail.
        self.tail = Idx::new_unchecked(new);

        // Update head index - only at first insertion.
        if self.head.option().is_none() {
            self.head = self.tail;
        }
    }

    /// Pushes an element at the end of the linked list.
    ///
    /// Complexity is `O(1)`.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// // The largest value will always be first
    /// ll.push_back(1).unwrap();
    /// assert_eq!(ll.back(), Some(&1));
    ///
    /// ll.push_back(2).unwrap();
    /// assert_eq!(ll.back(), Some(&2));
    ///
    /// ll.push_back(3).unwrap();
    /// assert_eq!(ll.back(), Some(&3));
    ///
    /// // This will not fit in the queue.
    /// assert_eq!(ll.push_back(4), Err(4));
    /// ```
    pub fn push_back(&mut self, value: T) -> Result<(), T> {
        if !self.is_full() {
            Ok(unsafe {
                self.push_back_unchecked(value);
            })
        } else {
            Err(value)
        }
    }

    /// Get an iterator over the list.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// ll.push_front(1).unwrap();
    /// ll.push_front(2).unwrap();
    ///
    /// let mut iter = ll.iter();
    ///
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<'_, T, Idx, N> {
        let done = self.is_empty();
        Iter {
            list: self,
            front: self.head,
            back: self.tail,
            done,
        }
    }

    /// Get a mutable iterator over the list.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// ll.push_front(1).unwrap();
    /// ll.push_front(2).unwrap();
    ///
    /// let mut iter = ll.iter_mut();
    ///
    /// assert_eq!(iter.next(), Some(&mut 2));
    /// assert_eq!(iter.next(), Some(&mut 1));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<'_, T, Idx, N> {
        let done = self.is_empty();
        let front = self.head;
        let back = self.tail;

        IterMut {
            list: self,
            front,
            back,
            done,
        }
    }

    /// Find an element in the list that can be changed.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// ll.push_back(1).unwrap();
    /// ll.push_back(2).unwrap();
    /// ll.push_back(3).unwrap();
    ///
    /// // Find a value and update it
    /// let mut find = ll.find_mut(|v| *v == 2).unwrap();
    /// *find += 1000;
    ///
    /// assert_eq!(ll.pop_back(), Ok(3));
    /// assert_eq!(ll.pop_back(), Ok(1002));
    /// assert_eq!(ll.pop_back(), Ok(1));
    /// assert_eq!(ll.pop_back(), Err(()));
    /// ```
    pub fn find_mut<F>(&mut self, mut f: F) -> Option<FindMut<'_, T, Idx, N>>
    where
        F: FnMut(&mut T) -> bool,
    {
        let head = self.head.option()?;

        // Special-case, first element
        if f(self.read_mut_data_in_node_at(head)) {
            return Some(FindMut {
                is_head: true,
                prev_index: Idx::none(),
                index: self.head,
                list: self,
            });
        }

        let mut current = head;

        while let Some(next) = self.node_at(current).next.option() {
            if f(self.read_mut_data_in_node_at(next)) {
                return Some(FindMut {
                    is_head: false,
                    prev_index: unsafe { Idx::new_unchecked(current) },
                    index: unsafe { Idx::new_unchecked(next) },
                    list: self,
                });
            }

            current = next;
        }

        None
    }

    /// Retains only the elements specified by the predicate `f`.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the
    /// original order, and preserves the order of the retained elements.
    ///
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.retain_mut(|elem| f(elem));
    }

    /// Retains only the elements specified by the predicate `f`, passing a mutable reference to it.
    ///
    /// In other words, remove all elements `e` such that `f(&mut e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the
    /// original order, and preserves the order of the retained elements.
    ///
    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        let mut index = self.head;
        while let Some(i) = index.option() {
            let node = self.node_at(i);
            let next = node.next;

            if !f(self.read_mut_data_in_node_at(i)) {
                // Re-point the previous index
                if let Some(prev) = self.node_at(i).prev.option() {
                    self.node_at_mut(prev).next = self.node_at(i).next;
                } else {
                    // Re-point head cursor
                    self.head = self.node_at(i).next;
                }

                // Re-point the next index
                if let Some(next) = self.node_at(i).next.option() {
                    self.node_at_mut(next).prev = self.node_at(i).prev;
                } else {
                    // Re-point tail cursor
                    self.tail = self.node_at(i).prev;
                }

                // Release the index into the free queue
                self.node_at_mut(i).next = self.free;
                self.free = index;

                // Decrement len
                self.len -= 1;

                //self.extract_data_in_node_at(i);
            }

            index = next;
        }
    }

    /// Provides a reference to the front element, or `None` if the list is
    /// empty.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// ll.push_front(1).unwrap();
    /// assert_eq!(ll.front(), Some(&1));
    /// ll.push_front(2).unwrap();
    /// assert_eq!(ll.front(), Some(&2));
    /// ll.push_front(3).unwrap();
    /// assert_eq!(ll.front(), Some(&3));
    /// ```
    pub fn front(&self) -> Option<&T> {
        self.head
            .option()
            .map(|head| self.read_data_in_node_at(head))
    }

    /// Provides a mutable reference to the front element, or `None` if the list
    /// is empty.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// assert_eq!(ll.front(), None);
    ///
    /// ll.push_front(1);
    /// assert_eq!(ll.front(), Some(&1));
    ///
    /// match ll.front_mut() {
    ///     None => {},
    ///     Some(x) => *x = 5,
    /// }
    /// assert_eq!(ll.front(), Some(&5));
    /// ```
    pub fn front_mut(&mut self) -> Option<&mut T> {
        if let Some(head) = self.head.option() {
            Some(self.read_mut_data_in_node_at(head))
        } else {
            None
        }
    }

    /// Provides a reference to the back element, or `None` if the list is
    /// empty.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// ll.push_back(1).unwrap();
    /// assert_eq!(ll.back(), Some(&1));
    /// ll.push_back(2).unwrap();
    /// assert_eq!(ll.back(), Some(&2));
    /// ll.push_back(3).unwrap();
    /// assert_eq!(ll.back(), Some(&3));
    /// ```
    pub fn back(&self) -> Option<&T> {
        self.tail
            .option()
            .map(|tail| self.read_data_in_node_at(tail))
    }

    /// Provides a mutable reference to the back element, or `None` if the list
    /// is empty.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// assert_eq!(ll.back(), None);
    ///
    /// ll.push_back(1);
    /// assert_eq!(ll.back(), Some(&1));
    ///
    /// match ll.back_mut() {
    ///     None => {},
    ///     Some(x) => *x = 5,
    /// }
    /// assert_eq!(ll.back(), Some(&5));
    /// ```
    pub fn back_mut(&mut self) -> Option<&mut T> {
        if let Some(tail) = self.tail.option() {
            Some(self.read_mut_data_in_node_at(tail))
        } else {
            None
        }
    }

    /// Pop the first element from the list without checking so the list is not empty.
    ///
    /// # Safety
    ///
    /// Assumes that the list is not empty.
    pub unsafe fn pop_front_unchecked(&mut self) -> T {
        let head = self.head.get_unchecked();
        let current = head;

        // Update 'head' cursor.
        // 'prev' cursor is none since we are at the beginning of the list.
        self.head = self.node_at(head).next;
        if let Some(new_head) = self.head.option() {
            self.node_at_mut(new_head).prev = Idx::none();
        } else {
            // List is empty now, set tail to None.
            self.tail = Idx::none();
        }

        // Update 'free' cursor.
        // 'current' element becomes the first 'free' element.
        self.node_at_mut(current).next = self.free;
        self.free = Idx::new_unchecked(current);

        // Decrement len
        self.len -= 1;

        // Return node content.
        self.extract_data_in_node_at(current)
    }

    /// Pop the last from the list without checking so the list is not empty.
    ///
    /// # Safety
    ///
    /// Assumes that the list is not empty.
    pub unsafe fn pop_back_unchecked(&mut self) -> T {
        let tail = self.tail.get_unchecked();
        let current = tail;

        // Update 'tail' cursor.
        // 'next' cursor is none since we are at the end of the list.
        self.tail = self.node_at(tail).prev;
        if let Some(new_tail) = self.tail.option() {
            self.node_at_mut(new_tail).next = Idx::none();
        } else {
            // List is empty now, set head to None.
            self.head = Idx::none();
        }

        // Update 'free' cursor.
        // 'current' element becomes the first 'free' element.
        self.node_at_mut(current).next = self.free;
        self.free = Idx::new_unchecked(current);

        // Decrement len
        self.len -= 1;

        self.extract_data_in_node_at(current)
    }

    /// Pops the first element in the list.
    ///
    /// Complexity is worst-case `O(1)`.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// ll.push_front(1).unwrap();
    /// ll.push_front(2).unwrap();
    ///
    /// assert_eq!(ll.pop_front(), Ok(2));
    /// assert_eq!(ll.pop_front(), Ok(1));
    /// assert_eq!(ll.pop_front(), Err(()));
    /// ```
    pub fn pop_front(&mut self) -> Result<T, ()> {
        if !self.is_empty() {
            Ok(unsafe { self.pop_front_unchecked() })
        } else {
            Err(())
        }
    }

    /// Pops the last element in the list.
    ///
    /// Complexity is worst-case `O(1)`.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// ll.push_back(1).unwrap();
    /// ll.push_back(2).unwrap();
    ///
    /// assert_eq!(ll.pop_back(), Ok(2));
    /// assert_eq!(ll.pop_back(), Ok(1));
    /// assert_eq!(ll.pop_back(), Err(()));
    /// ```
    pub fn pop_back(&mut self) -> Result<T, ()> {
        if !self.is_empty() {
            Ok(unsafe { self.pop_back_unchecked() })
        } else {
            Err(())
        }
    }

    /// Checks if the linked list is full.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// assert_eq!(ll.is_full(), false);
    ///
    /// ll.push_back(1).unwrap();
    /// assert_eq!(ll.is_full(), false);
    /// ll.push_back(2).unwrap();
    /// assert_eq!(ll.is_full(), false);
    /// ll.push_back(3).unwrap();
    /// assert_eq!(ll.is_full(), true);
    /// ```
    #[inline]
    pub fn is_full(&self) -> bool {
        self.free.option().is_none()
    }

    /// Checks if the linked list is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// assert_eq!(ll.is_empty(), true);
    ///
    /// ll.push_back(1).unwrap();
    /// assert_eq!(ll.is_empty(), false);
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.head.option().is_none()
    }

    /// Removes all the linked list nodes.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// ll.push_back(1).unwrap();
    /// assert_eq!(ll.is_empty(), false);
    ///
    /// ll.clear();
    /// assert_eq!(ll.is_empty(), true);
    /// ```
    pub fn clear(&mut self) {
        self.retain(|_| false)
    }
}

/// An iterator for the linked list.
pub struct Iter<'a, T, Idx, const N: usize>
where
    Idx: LinkedListIndex,
{
    list: &'a LinkedList<T, Idx, N>,
    front: Idx,
    back: Idx,
    done: bool,
}

impl<'a, T, Idx, const N: usize> Iterator for Iter<'a, T, Idx, N>
where
    Idx: LinkedListIndex,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            None
        } else {
            if self.front == self.back {
                self.done = true;
            }

            let index = self.front.option()?;
            self.front = self.list.node_at(index).next;

            Some(unsafe { &*self.list.node_at(index).val.as_ptr() })
        }
    }
}

impl<'a, T, Idx, const N: usize> DoubleEndedIterator for Iter<'a, T, Idx, N>
where
    Idx: LinkedListIndex,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.done {
            None
        } else {
            if self.front == self.back {
                self.done = true;
            }

            let index = self.back.option()?;
            self.back = self.list.node_at(index).next;

            Some(unsafe { &*self.list.node_at(index).val.as_ptr() })
        }
    }
}

/// A mutable iterator for the linked list.
pub struct IterMut<'a, T, Idx, const N: usize>
where
    Idx: LinkedListIndex,
{
    list: &'a mut LinkedList<T, Idx, N>,
    front: Idx,
    back: Idx,
    done: bool,
}

impl<'a, T, Idx, const N: usize> Iterator for IterMut<'a, T, Idx, N>
where
    Idx: LinkedListIndex,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            None
        } else {
            if self.front == self.back {
                self.done = true;
            }

            let index = self.front.option()?;
            self.front = self.list.node_at(index).next;

            Some(unsafe { &mut *self.list.node_at_mut(index).val.as_mut_ptr() })
        }
    }
}

impl<'a, T, Idx, const N: usize> DoubleEndedIterator for IterMut<'a, T, Idx, N>
where
    Idx: LinkedListIndex,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.done {
            None
        } else {
            if self.front == self.back {
                self.done = true;
            }

            let index = self.back.option()?;
            self.back = self.list.node_at(index).next;

            Some(unsafe { &mut *self.list.node_at_mut(index).val.as_mut_ptr() })
        }
    }
}

impl<'a, T, Idx, const N: usize> IntoIterator for &'a LinkedList<T, Idx, N>
where
    Idx: LinkedListIndex,
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T, Idx, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, Idx, const N: usize> IntoIterator for &'a mut LinkedList<T, Idx, N>
where
    Idx: LinkedListIndex,
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T, Idx, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Comes from [`LinkedList::find_mut`].
pub struct FindMut<'a, T, Idx, const N: usize>
where
    Idx: LinkedListIndex,
{
    list: &'a mut LinkedList<T, Idx, N>,
    is_head: bool,
    prev_index: Idx,
    index: Idx,
}

impl<'a, T, Idx, const N: usize> FindMut<'a, T, Idx, N>
where
    Idx: LinkedListIndex,
{
    fn pop_internal(&mut self) -> T {
        if self.is_head {
            // If it is the head element, we can do a pop_front.
            unsafe { self.list.pop_front_unchecked() }
        } else {
            // Somewhere in the list
            let prev = unsafe { self.prev_index.get_unchecked() };
            let curr = unsafe { self.index.get_unchecked() };

            // Re-point the previous index
            self.list.node_at_mut(prev).next = self.list.node_at_mut(curr).next;

            // Re-point the next index
            if let Some(next) = self.list.node_at_mut(curr).next.option() {
                self.list.node_at_mut(next).prev = self.list.node_at_mut(curr).prev;
            } else {
                // Re-point tail cursor
                self.list.tail = self.list.node_at(curr).prev;
            }

            // Release the index into the free queue
            self.list.node_at_mut(curr).next = self.list.free;
            self.list.free = self.index;

            // Decrement len
            self.list.len -= 1;

            self.list.extract_data_in_node_at(curr)
        }
    }

    /// This will pop the element from the list.
    ///
    /// Complexity is worst-case `O(1)`.
    ///
    /// # Example
    ///
    /// ```
    /// use heapless::linked_list::LinkedList;
    /// let mut ll: LinkedList<_, _, 3> = LinkedList::new_usize();
    ///
    /// ll.push_front(1).unwrap();
    /// ll.push_front(2).unwrap();
    /// ll.push_front(3).unwrap();
    ///
    /// // Find a value and update it
    /// let mut find = ll.find_mut(|v| *v == 2).unwrap();
    /// find.pop();
    ///
    /// assert_eq!(ll.pop_front(), Ok(3));
    /// assert_eq!(ll.pop_front(), Ok(1));
    /// assert_eq!(ll.pop_front(), Err(()));
    /// ```
    #[inline]
    pub fn pop(mut self) -> T {
        self.pop_internal()
    }
}

impl<T, Idx, const N: usize> Deref for FindMut<'_, T, Idx, N>
where
    Idx: LinkedListIndex,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.list
            .read_data_in_node_at(unsafe { self.index.get_unchecked() })
    }
}

impl<T, Idx, const N: usize> DerefMut for FindMut<'_, T, Idx, N>
where
    Idx: LinkedListIndex,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.list
            .read_mut_data_in_node_at(unsafe { self.index.get_unchecked() })
    }
}

/// Useful for debug during development.
// impl<T, Idx, const N: usize> fmt::Debug for FindMut<'_, T, Idx, N>
// where
//     T: core::fmt::Debug,
//     Idx: LinkedListIndex,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_struct("FindMut")
//             .field("prev_index", &self.prev_index.option())
//             .field("index", &self.index.option())
//             .field(
//                 "prev_value",
//                 &self
//                     .list
//                     .read_data_in_node_at(self.prev_index.option().unwrap()),
//             )
//             .field(
//                 "value",
//                 &self.list.read_data_in_node_at(self.index.option().unwrap()),
//             )
//             .finish()
//     }
// }

impl<T, Idx, const N: usize> fmt::Debug for LinkedList<T, Idx, N>
where
    T: core::fmt::Debug,
    Idx: LinkedListIndex,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T, Idx, const N: usize> Drop for LinkedList<T, Idx, N>
where
    Idx: LinkedListIndex,
{
    fn drop(&mut self) {
        let mut index = self.head;

        while let Some(i) = index.option() {
            let node = self.node_at_mut(i);
            index = node.next;

            unsafe {
                ptr::drop_in_place(node.val.as_mut_ptr());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_new() {
        static mut _V1: LinkedList<u32, LinkedIndexU8, 100> = LinkedList::new_u8();
        static mut _V2: LinkedList<u32, LinkedIndexU16, 10_000> = LinkedList::new_u16();
        static mut _V3: LinkedList<u32, LinkedIndexUsize, 100_000> = LinkedList::new_usize();
    }

    #[test]
    fn test_drop() {
        droppable!();

        {
            let mut ll: LinkedList<Droppable, LinkedIndexUsize, 2> = LinkedList::new_usize();
            ll.push_back(Droppable::new()).ok().unwrap();
            ll.push_back(Droppable::new()).ok().unwrap();
            ll.pop_front().unwrap();
            assert_eq!(Droppable::count(), 1);
        }

        {
            let mut ll: LinkedList<Droppable, LinkedIndexUsize, 2> = LinkedList::new_usize();
            ll.push_back(Droppable::new()).ok().unwrap();
            ll.push_back(Droppable::new()).ok().unwrap();
        }

        assert_eq!(Droppable::count(), 0);
        {
            let mut ll: LinkedList<Droppable, LinkedIndexUsize, 2> = LinkedList::new_usize();
            ll.push_front(Droppable::new()).ok().unwrap();
            ll.push_front(Droppable::new()).ok().unwrap();
        }

        assert_eq!(Droppable::count(), 0);
    }

    #[test]
    fn test_front() {
        let mut ll: LinkedList<u32, LinkedIndexUsize, 3> = LinkedList::new_usize();

        ll.push_front(1).unwrap();
        assert_eq!(ll.front().unwrap(), &1);

        ll.push_front(2).unwrap();
        assert_eq!(ll.front().unwrap(), &2);

        ll.push_front(3).unwrap();
        assert_eq!(ll.front().unwrap(), &3);

        let mut ll: LinkedList<u32, LinkedIndexUsize, 3> = LinkedList::new_usize();

        ll.push_back(2).unwrap();
        assert_eq!(ll.back().unwrap(), &2);

        ll.push_back(1).unwrap();
        assert_eq!(ll.back().unwrap(), &1);

        ll.push_back(3).unwrap();
        assert_eq!(ll.back().unwrap(), &3);
    }

    #[test]
    fn test_full() {
        let mut ll: LinkedList<u32, LinkedIndexUsize, 4> = LinkedList::new_usize();
        ll.push_front(0).unwrap();
        ll.push_back(1).unwrap();
        ll.push_back(2).unwrap();
        ll.push_back(3).unwrap();

        assert!(ll.push_front(4).is_err());
        assert!(ll.push_back(4).is_err());
        assert!(ll.is_full())
    }

    #[test]
    fn test_empty() {
        let mut ll: LinkedList<u32, LinkedIndexUsize, 3> = LinkedList::new_usize();

        assert!(ll.is_empty());

        ll.push_back(0).unwrap();
        assert!(!ll.is_empty());

        ll.push_front(1).unwrap();
        assert!(!ll.is_empty());

        ll.pop_front().unwrap();
        ll.pop_front().unwrap();

        assert!(ll.pop_front().is_err());
        assert!(ll.pop_back().is_err());
        assert!(ll.is_empty());
    }

    #[test]
    fn test_zero_size() {
        let ll: LinkedList<u32, LinkedIndexUsize, 0> = LinkedList::new_usize();

        assert!(ll.is_empty());
        assert!(ll.is_full());
    }

    #[test]
    fn test_rejected_push() {
        let mut ll: LinkedList<u32, LinkedIndexUsize, 3> = LinkedList::new_usize();
        ll.push_back(1).unwrap();
        ll.push_back(2).unwrap();
        ll.push_back(3).unwrap();

        // This won't fit
        let r = ll.push_back(4);

        assert_eq!(r, Err(4));
    }

    #[test]
    fn test_updating() {
        let mut ll: LinkedList<u32, LinkedIndexUsize, 3> = LinkedList::new_usize();
        ll.push_front(1).unwrap();
        ll.push_front(2).unwrap();
        ll.push_front(3).unwrap();

        let mut find = ll.find_mut(|v| *v == 2).unwrap();

        *find += 1000;

        let find = ll.find_mut(|v| *v == 1002).unwrap();
        assert_eq!(*find, 1002);

        let mut find = ll.find_mut(|v| *v == 3).unwrap();

        *find += 1000;

        let find = ll.find_mut(|v| *v == 1003).unwrap();
        assert_eq!(*find, 1003);
    }

    #[test]
    fn test_updating_1() {
        let mut ll: LinkedList<u32, LinkedIndexUsize, 3> = LinkedList::new_usize();
        ll.push_front(1).unwrap();

        let v = ll.pop_front().unwrap();

        assert_eq!(v, 1);
    }

    #[test]
    fn test_updating_2() {
        let mut ll: LinkedList<u32, LinkedIndexUsize, 3> = LinkedList::new_usize();
        ll.push_front(1).unwrap();

        let mut find = ll.find_mut(|v| *v == 1).unwrap();

        *find += 1000;

        assert_eq!(ll.front().unwrap(), &1001);
    }

    #[test]
    fn test_updating_3() {
        let mut ll: LinkedList<u32, LinkedIndexUsize, 6> = LinkedList::new_usize();
        ll.push_back(1).unwrap();
        ll.push_back(2).unwrap();
        ll.push_back(3).unwrap();
        ll.push_back(4).unwrap();
        ll.push_back(5).unwrap();

        ll.pop_front().unwrap();
        ll.pop_back().unwrap();

        assert_eq!(ll.front().unwrap(), &2);
        assert_eq!(ll.back().unwrap(), &4);

        ll.pop_back().unwrap();
        ll.push_back(55).unwrap();
        ll.push_front(0).unwrap();

        assert_eq!(ll.front().unwrap(), &0);
        assert_eq!(ll.back().unwrap(), &55);
    }

    #[test]
    fn test_front_back() {
        let mut ll: LinkedList<i32, LinkedIndexUsize, 4> = LinkedList::new_usize();
        assert_eq!(ll.front(), None);
        assert_eq!(ll.front_mut(), None);
        assert_eq!(ll.back(), None);
        assert_eq!(ll.back_mut(), None);

        ll.push_back(4).unwrap();
        assert_eq!(ll.front(), Some(&4));
        assert_eq!(ll.front_mut(), Some(&mut 4));
        assert_eq!(ll.back(), Some(&4));
        assert_eq!(ll.back_mut(), Some(&mut 4));

        ll.push_front(3).unwrap();
        assert_eq!(ll.front(), Some(&3));
        assert_eq!(ll.front_mut(), Some(&mut 3));
        assert_eq!(ll.back(), Some(&4));
        assert_eq!(ll.back_mut(), Some(&mut 4));

        ll.pop_back().unwrap();
        assert_eq!(ll.front(), Some(&3));
        assert_eq!(ll.front_mut(), Some(&mut 3));
        assert_eq!(ll.back(), Some(&3));
        assert_eq!(ll.back_mut(), Some(&mut 3));

        ll.pop_front().unwrap();
        assert_eq!(ll.front(), None);
        assert_eq!(ll.front_mut(), None);
        assert_eq!(ll.back(), None);
        assert_eq!(ll.back_mut(), None);
    }

    #[test]
    fn test_iter() {
        let mut ll: LinkedList<i32, LinkedIndexUsize, 4> = LinkedList::new_usize();

        ll.push_back(0).unwrap();
        ll.push_back(1).unwrap();
        ll.push_front(2).unwrap();
        ll.push_front(3).unwrap();
        ll.pop_back().unwrap();
        ll.push_front(4).unwrap();

        let mut items = ll.iter();

        assert_eq!(items.next(), Some(&4));
        assert_eq!(items.next(), Some(&3));
        assert_eq!(items.next(), Some(&2));
        assert_eq!(items.next(), Some(&0));
        assert_eq!(items.next(), None);
    }

    #[test]
    fn test_iter_mut() {
        let mut ll: LinkedList<i32, LinkedIndexUsize, 4> = LinkedList::new_usize();

        ll.push_back(0).unwrap();
        ll.push_back(1).unwrap();
        ll.push_front(2).unwrap();
        ll.push_front(3).unwrap();
        ll.pop_back().unwrap();
        ll.push_front(4).unwrap();

        let mut items = ll.iter_mut();

        assert_eq!(items.next(), Some(&mut 4));
        assert_eq!(items.next(), Some(&mut 3));
        assert_eq!(items.next(), Some(&mut 2));
        assert_eq!(items.next(), Some(&mut 0));
        assert_eq!(items.next(), None);
    }

    #[test]
    fn test_iter_move() {
        let mut ll: LinkedList<i32, LinkedIndexUsize, 4> = LinkedList::new_usize();
        ll.push_back(0).unwrap();
        ll.push_back(1).unwrap();
        ll.push_back(2).unwrap();
        ll.push_back(3).unwrap();

        let mut items = ll.into_iter();

        assert_eq!(items.next(), Some(&0));
        assert_eq!(items.next(), Some(&1));
        assert_eq!(items.next(), Some(&2));
        assert_eq!(items.next(), Some(&3));
        assert_eq!(items.next(), None);
    }

    #[test]
    fn test_iter_move_drop() {
        droppable!();

        {
            let mut ll: LinkedList<Droppable, LinkedIndexUsize, 2> = LinkedList::new_usize();
            ll.push_back(Droppable::new()).ok().unwrap();
            ll.push_back(Droppable::new()).ok().unwrap();
            let mut items = ll.into_iter();
            // Move all
            let _ = items.next();
            let _ = items.next();
        }

        assert_eq!(Droppable::count(), 0);

        {
            let mut ll: LinkedList<Droppable, LinkedIndexUsize, 2> = LinkedList::new_usize();
            ll.push_back(Droppable::new()).ok().unwrap();
            ll.push_back(Droppable::new()).ok().unwrap();
            let _items = ll.into_iter();
            // Move none
        }

        assert_eq!(Droppable::count(), 0);

        {
            let mut ll: LinkedList<Droppable, LinkedIndexUsize, 2> = LinkedList::new_usize();
            ll.push_back(Droppable::new()).ok().unwrap();
            ll.push_back(Droppable::new()).ok().unwrap();
            let mut items = ll.into_iter();
            let _ = items.next(); // Move partly
        }

        assert_eq!(Droppable::count(), 0);
    }

    #[test]
    fn test_push_and_pop() {
        let mut ll: LinkedList<i32, LinkedIndexUsize, 4> = LinkedList::new_usize();
        assert_eq!(ll.len(), 0);

        assert_eq!(ll.pop_front(), Err(()));
        assert_eq!(ll.pop_back(), Err(()));
        assert_eq!(ll.len(), 0);

        ll.push_back(0).unwrap();
        assert_eq!(ll.len(), 1);

        assert_eq!(ll.pop_back(), Ok(0));
        assert_eq!(ll.len(), 0);

        ll.push_back(0).unwrap();
        ll.push_back(1).unwrap();
        ll.push_front(2).unwrap();
        ll.push_front(3).unwrap();
        assert_eq!(ll.len(), 4);

        // List contains: 3 2 0 1
        assert_eq!(ll.pop_front(), Ok(3));
        assert_eq!(ll.len(), 3);
        assert_eq!(ll.pop_front(), Ok(2));
        assert_eq!(ll.len(), 2);
        assert_eq!(ll.pop_back(), Ok(1));
        assert_eq!(ll.len(), 1);
        assert_eq!(ll.pop_front(), Ok(0));
        assert_eq!(ll.len(), 0);

        // List is now empty
        assert_eq!(ll.pop_front(), Err(()));
        assert_eq!(ll.pop_back(), Err(()));
        assert_eq!(ll.len(), 0);
    }

    #[test]
    fn test_retain() {
        let mut ll: LinkedList<i32, LinkedIndexUsize, 4> = LinkedList::new_usize();

        ll.push_back(1).unwrap();
        ll.push_back(2).unwrap();
        ll.push_back(3).unwrap();
        ll.push_back(4).unwrap();

        ll.retain(|val| val % 2 != 0);

        assert_eq!(ll.pop_back(), Ok(3));
        assert_eq!(ll.pop_back(), Ok(1));
    }

    #[test]
    fn test_retain_mut() {
        let mut ll: LinkedList<i32, LinkedIndexUsize, 4> = LinkedList::new_usize();

        ll.push_back(1).unwrap();
        ll.push_back(2).unwrap();
        ll.push_back(3).unwrap();
        ll.push_back(4).unwrap();

        ll.retain_mut(|val| {
            *val += 1;
            *val % 2 == 0
        });

        assert_eq!(ll.pop_back(), Ok(4));
        assert_eq!(ll.pop_back(), Ok(2));
    }
}
